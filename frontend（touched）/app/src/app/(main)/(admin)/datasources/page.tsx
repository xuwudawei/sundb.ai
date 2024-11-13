import { AdminPageHeading } from '@/components/admin-page-heading';
import { DatasourceTable } from '@/components/datasource/DatasourceTable';

export default function ChatEnginesPage () {
  return (
    <>
      <AdminPageHeading title="Datasources" />
      <DatasourceTable />
    </>
  );
}
